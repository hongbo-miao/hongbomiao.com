import Website from '../types/Website';
import './SocialList.css';

type Props = {
  websites: ReadonlyArray<Website>;
};

function SocialList(props: Props) {
  const { websites } = props;
  return (
    <>
      {websites.map((website) => {
        const { name, src, url } = website;
        return (
          <div key={name} className="inline-flex items-center">
            <a href={url} target="_blank" rel="noopener noreferrer">
              <img className="hm-logo" src={src} height="32" width="32" alt={name} />
            </a>
          </div>
        );
      })}
    </>
  );
}

export default SocialList;
